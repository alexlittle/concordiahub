import pandas as pd
import ollama
import json
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from examples import config

COLUMN_MAPPINGS = {
    'sex': { 0: 'Female', 1: 'Male'},
    'chest pain type': { 1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal pain', 4: 'asymptomatic'},
    'fasting blood sugar': { 0: 'false', 1: 'true'},
    'resting ECG': { 0: 'normal', 1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy'},
    'exercise induced angina': { 0: 'no', 1: 'yes'},
    'slope of the peak exercise': { 1: 'upsloping', 2: 'flat', 3: 'downsloping'},
    'thalassemia': { 3: 'normal', 6: 'fixed defect', 7: 'reversible defect'}
}

df = pd.read_csv(config.HEART_INPUT_FILE)

for col, mapping in COLUMN_MAPPINGS.items():
    df[col] = df[col].map(mapping)

print(df.head())

jinja_env = Environment(
    loader=FileSystemLoader(config.LLM_TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True
)

prompt_template = jinja_env.get_template("heart.jinja")

def run_llm(prompt):
    response = ollama.chat(
        model=config.HEART_OLLAMA_MODEL,
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

        heart_disease = data.get("heart_disease", 0)
        explanation = data.get("explanation", "")

    except json.JSONDecodeError:
        # if invalid JSON
        heart_disease = 0
        explanation = ""

    return {"llm_heart_disease": heart_disease,
        "llm_explanation": explanation,
        "raw_output": text }

outputs = []

for idx, row in df[:3].iterrows():
    print(f"Running LLM on conversation {idx + 1}/{len(df)} ")

    prompt = prompt_template.render(
        age=row['age'],
        sex=row['sex'],
        chest_pain_type=row['chest pain type'],
        resting_blood_pressure=row['resting blood pressure'],
        serum_cholesterol=row['serum cholesterol'],
        fasting_blood_sugar=row['fasting blood sugar'],
        resting_ECG=row['resting ECG'],
        maximum_heart_rate=row['maximum heart rate'],
        exercise_induced_angina=row['exercise induced angina'],
        ST_depression=row['ST depression'],
        slope_of_the_peak_exercise=row['slope of the peak exercise'],
        number_of_major_vessels=row['number of major vessels'],
        thalassemia=row['thalassemia'])

    llm_text = run_llm(prompt)
    parsed_output = parse_output(llm_text)

    outputs.append({
        "sample_uuid": row[config.SAMPLE_UUID_COL],
        "target": row["target"],
        "prediction": parsed_output["llm_heart_disease"],
        "cor_explanation": parsed_output["llm_explanation"],
    })

out_df = pd.DataFrame(outputs)

X = df.drop(columns=["target"])
y = df[config.SAMPLE_UUID_COL, "target"]
y_preds = out_df[config.SAMPLE_UUID_COL, "prediction"]
y_exp = out_df[config.SAMPLE_UUID_COL, "cor_explanation"]

X.to_parquet('./cache/heart/llm/X.parquet')
y.to_parquet('./cache/heart/llm/y.parquet')
y_preds.to_frame().to_parquet('./cache/heart/llm/y_preds.parquet')
y_exp.to_frame().to_parquet('./cache/heart/llm/y_exp.parquet')