import pandas as pd
import ollama
import json
import os

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from examples import config
from examples.helpers import safe_dirname

# -----------------------------
# Load input conversations
# -----------------------------
df = pd.read_csv(config.MEDSYNTH_FULL_FILE)

# only run on those that match the test samples from XGboost model
xgboost_df = pd.read_parquet('./cache/medsynth/xgboost/y_test.parquet')

df = df[df['sample_uuid'].isin(xgboost_df['sample_uuid'])]

# -----------------------------
# Prompt template
# -----------------------------

JINJA_ENV = Environment(
    loader=FileSystemLoader(config.LLM_TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True
)

def build_prompt(patient_data:str):
    user_prompt_template = JINJA_ENV.get_template("medsynth.jinja")
    return user_prompt_template.render(patient_data=patient_data)

# run Ollama
def run_llm_request(patient_data: dict, model: str):
    system_prompt_template = JINJA_ENV.get_template("mdesynth_system.jinja")
    system_prompt = system_prompt_template.render()

    response = ollama.chat(
        model=config.HEART_OLLAMA_MODEL,
        stream=False,
        format="json",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_prompt(patient_data)}
        ],
        logprobs=True,
        options={
            "num_ctx": 8192,
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

# Run LLM over scenarios
pred_outputs = []
exp_outputs = []

for idx, row in df.head(1).iterrows():
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
            "explanation": json.dumps(response, default=str)
        }
        pred_outputs.append(pred_output)
        exp_outputs.append(exp_output)
        print(
            f"Target: {row["target"]}, Prediction: {pred_output['prediction']}")
        print(f"Explanation: {exp_output['explanation']}")
    else:
        exp_output = {
            "sample_uuid": row["sample_uuid"],
            "explanation": response
        }
        exp_outputs.append(exp_output)

# Save outputs
pred_outputs_df = pd.DataFrame(pred_outputs)
exp_outputs_df = pd.DataFrame(exp_outputs)

y = df[[config.SAMPLE_UUID_COL, "target"]]
X = df.drop(columns=["target"])


dir_name = f"llm_{safe_dirname(config.HEART_OLLAMA_MODEL)}"
os.makedirs(f"./cache/medsynth/{dir_name}/", exist_ok=True)

X.to_parquet(f"./cache/medsynth/{dir_name}/X.parquet")
y.to_parquet(f"./cache/medsynth/{dir_name}/y.parquet")
pred_outputs_df.to_parquet(f"./cache/medsynth/{dir_name}/y_preds.parquet")
exp_outputs_df.to_parquet(f"./cache/medsynth/{dir_name}/y_exp.parquet")
