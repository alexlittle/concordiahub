import pandas as pd
import ollama
import re
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

# Helper: parse output
def parse_output(text: str) -> dict:
    """
    Extract Diagnosis and Explanation sections.
    Falls back gracefully if formatting is imperfect.
    """
    diagnosis = ""
    explanation = ""

    diag_match = re.search(r"Diagnosis:\s*(.+)", text, re.IGNORECASE)
    expl_match = re.search(r"Explanation:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    if diag_match:
        diagnosis = diag_match.group(1).strip()

    if expl_match:
        explanation = expl_match.group(1).strip()
    else:
        # fallback: everything after diagnosis
        explanation = text

    return {
        "llm_diagnosis": diagnosis,
        "llm_explanation": explanation,
        "raw_output": text
    }

# Run LLM over conversations
outputs = []

for idx, row in df.head(10).iterrows():
    print(f"Running LLM on conversation {idx + 1}/{len(df)} ")

    prompt = prompt_template.render(conversation=row.conversation_text)

    llm_text = run_llm(prompt)
    parsed = parse_output(llm_text)

    outputs.append({
        "id": idx,
        "true_label": row["true_label"],
        **parsed
    })

# Save outputs
out_df = pd.DataFrame(outputs)

out_df.to_csv(OUTPUT_DIR / "llm_outputs_full.csv", index=False)

# Convenience: explanation-only corpus
out_df[["id", "true_label", "llm_explanation"]].to_csv(
    OUTPUT_DIR / "llm_explanations_only.csv",
    index=False
)

print("LLM inference complete.")
print(f"Saved to {OUTPUT_DIR}")