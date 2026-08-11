"""
Script to report how many of the LLM reported features match those from the classical models

MedSynth
"""

import pandas as pd


# note identical for LR and XGBoost so only loading from LR here
input_classical = "../data_store/medsynth_baseline/exp_global_coefficients.csv"

# for the LLMs
input_gemma = "../data_store/medsynth_llm_gemma3_27b-it-qat/exp_global_llm_reasoning.csv"
input_qwen = "../data_store/medsynth_llm_qwen2.5_14b-instruct-q8_0/exp_global_llm_reasoning.csv"


df_classical = pd.read_csv(input_classical)
df_gemma = pd.read_csv(input_gemma)
df_qwen = pd.read_csv(input_qwen)

classical = set(df_classical["feature"].dropna().str.strip().str.lower())
gemma = set(df_gemma["feature"].dropna().str.strip().str.lower())
qwen = set(df_qwen["feature"].dropna().str.strip().str.lower())



matched_gemma = classical & gemma
gemma_only = gemma - classical
classical_only = classical - gemma

gemma_coverage = len(matched_gemma) / len(gemma_only) if gemma_only else 1
print(f"Gemma: {gemma_coverage} of {len(gemma)} features matched")


matched_qwen = classical & qwen
qwen_only = qwen - classical
classical_only = classical - qwen

qwen_coverage = len(matched_qwen) / len(qwen_only) if qwen_only else 1
print(f"Qwen: {qwen_coverage} of {len(qwen)} features matched")
