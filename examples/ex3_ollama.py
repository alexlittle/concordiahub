# %% [markdown]
# # Local Ollama model on MedSynth data
# %%
import helpers
import ollama
import pandas as pd
from tqdm import tqdm

# %%
df_ms = helpers.process_medsynth()
df = helpers.extract_vitalsigns_tocols(df_ms)

# %%
MODEL = "gemma3:latest"

def ollama_predict(prompt, model=MODEL):
    response = ollama.generate(model=model, prompt=prompt)
    return response["response"]


allowed = df["ICD_chapter"].unique().tolist()

def restrict_output(output):
    for a in allowed:
        if output.lower() == a.lower():
            return a
    return "UNKNOWN"


def build_prompt(row):
    return f"""
You are a medical classification model.

Classify the ICD chapter of this case.

Dialogue:
{row['Dialogue']}

Vitals:
BP: {row['BP_systolic']}/{row['BP_diastolic']}
HR: {row['Heart_Rate']}
RR: {row['Respiratory_Rate']}
Temp C: {row['temp_c']}
SpO2: {row['Oxygen_Saturation']}
Oxygen Device: {row['Oxygen_Device']}

Answer ONLY with the ICD chapter name from this list {allowed}.
"""



# %%
df_test = df[0:5]
# %%

preds = []

for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):

    prompt = build_prompt(row)
    llm_output = ollama_predict(prompt)
    print(f"True Label: {row['ICD_chapter']} Predicted Label: {llm_output.strip()}")
    preds.append(llm_output.strip())



true_labels = df_test["ICD_chapter"].astype(str).str.strip()
pred_labels = pd.Series(preds)
