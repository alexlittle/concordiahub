import pandas as pd
import ollama
import re
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "llama3.1:8b"
INPUT_FILE = "./output/llm/llm_input_conversations.csv"
OUTPUT_DIR = Path("./output/llm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load input conversations
# -----------------------------
df = pd.read_csv(INPUT_FILE)
# Expected columns:
# conversation_id, true_label, conversation_text

# -----------------------------
# Prompt template
# -----------------------------
PROMPT_TEMPLATE = """You are given a transcript of a patient–doctor conversation.

Task:
1. Predict the most likely ICD-10 chapter for this conversation.
2. Explain your reasoning based only on the information expressed in the conversation.

Important constraints:
- Do not use external medical knowledge.
- Do not invent symptoms or facts not stated.
- Base your explanation on specific phrases or observations in the dialogue.

Conversation:
{conversation}

Output format:
Diagnosis:
Explanation:
"""

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

for idx, row in df.iterrows():
    print(f"Running LLM on conversation {idx + 1}/{len(df)} "
          f"(ID={row['conversation_id']})")

    prompt = PROMPT_TEMPLATE.format(
        conversation=row["conversation_text"]
    )

    llm_text = run_llm(prompt)
    parsed = parse_output(llm_text)

    outputs.append({
        "conversation_id": row["conversation_id"],
        "true_label": row["true_label"],
        **parsed
    })

# Save outputs
out_df = pd.DataFrame(outputs)

out_df.to_csv(
    OUTPUT_DIR / "llm_outputs_full.csv",
    index=False
)

# Convenience: explanation-only corpus
out_df[["conversation_id", "true_label", "llm_explanation"]].to_csv(
    OUTPUT_DIR / "llm_explanations_only.csv",
    index=False
)

print("LLM inference complete.")
print(f"Saved to {OUTPUT_DIR}")