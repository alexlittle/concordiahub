"""
Script to report how many of the LLM reported features match those from the classical models

Cleveland dataset
"""

import pandas as pd


# note identical for LR and XGBoost so only loading from LR here
input_classical = "../data_store/heart_lr/exp_global_coefficients.csv"

# for the LLMs
input_llama = "../data_store/heart_llm_llama3.1_8b/exp_global_llm_reasoning.csv"
input_gemma = "../data_store/heart_llm_gemma3_27b-it-qat/exp_global_llm_reasoning.csv"
input_qwen = "../data_store/heart_llm_qwen2.5_14b-instruct-q8_0/exp_global_llm_reasoning.csv"


df_classical = pd.read_csv(input_classical)
df_llama = pd.read_csv(input_llama)
df_gemma = pd.read_csv(input_gemma)
df_qwen = pd.read_csv(input_qwen)

classical = set(df_classical["feature"].dropna())
llama = set(df_llama["feature"].dropna())
gemma = set(df_gemma["feature"].dropna())
qwen = set(df_qwen["feature"].dropna())


matched_llama = classical & llama
llama_only = llama - classical
classical_only = classical - llama

llama_coverage = len(matched_llama) / len(llama_only) if llama_only else 1
print(f"llama coverage: {llama_coverage}")


matched_gemma = classical & gemma
gemma_only = gemma - classical
classical_only = classical - gemma

gemma_coverage = len(matched_gemma) / len(gemma_only) if gemma_only else 1
print(f"gemma coverage: {gemma_coverage}")


matched_qwen = classical & qwen
qwen_only = qwen - classical
classical_only = classical - qwen

qwen_coverage = len(matched_qwen) / len(qwen_only) if qwen_only else 1
print(f"qwen coverage: {qwen_coverage}")
