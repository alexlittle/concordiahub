import pandas as pd
import ollama
import json
import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from examples import config


#----------------------------
# Set up
#----------------------------
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

JINJA_ENV = Environment(
    loader=FileSystemLoader(config.LLM_TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True
)

#----------------------------
# Functions
#----------------------------
def build_prompt(patient_data:dict):
    user_prompt_template = JINJA_ENV.get_template("heart.jinja")

    feature_block = "\n".join(
        f"  - {key}: {value}"
        for key, value in patient_data.items()
    )
    return user_prompt_template.render(patient_data=feature_block)

def run_llm_request(patient_data: dict, model: str):
    system_prompt_template = JINJA_ENV.get_template("heart_system.jinja")
    system_prompt = system_prompt_template.render()

    response = ollama.chat(
        model=config.HEART_OLLAMA_MODEL,
        stream=False,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_prompt(patient_data)}
        ],
        logprobs=True,
        options={
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 42,
        })
    return response


def parse_response(response: str) -> dict:
    content = response["message"]["content"].strip()

    try:
        return json.loads(content), True
    except json.JSONDecodeError as e:
        print("Model returned invalid JSON")
        return content, False


#----------------------------
# Run and save responses
#----------------------------
pred_outputs = []
exp_outputs = []

for idx, row in df.iterrows():
    print(f"Running LLM on conversation {idx + 1}/{len(df)} ")
    row_dict = row.to_dict()
    row_dict.pop("sample_uuid")
    row_dict.pop("target")

    llm_response = run_llm_request(row_dict, config.HEART_OLLAMA_MODEL)
    response, valid_json = parse_response(llm_response)
    if valid_json:
        pred_output = {
            "sample_uuid": row["sample_uuid"],
            "prediction": response["prediction"]
        }
        exp_output = {
            "sample_uuid": row["sample_uuid"],
            "explanation": response
        }
        pred_outputs.append(pred_output)
        exp_outputs.append(exp_output)
    else:
        exp_output = {
            "sample_uuid": row["sample_uuid"],
            "explanation": response
        }
        exp_outputs.append(exp_output)



pred_outputs_df = pd.DataFrame(pred_outputs)
exp_outputs_df = pd.DataFrame(exp_outputs)

y = df[[config.SAMPLE_UUID_COL, "target"]]
X = df.drop(columns=["target"])

os.makedirs("./cache/heart/llm/", exist_ok=True)
X.to_parquet('./cache/heart/llm/X.parquet')
y.to_parquet('./cache/heart/llm/y.parquet')
pred_outputs_df.to_parquet('./cache/heart/llm/y_preds.parquet')
exp_outputs_df.to_parquet('./cache/heart/llm/y_exp.parquet')