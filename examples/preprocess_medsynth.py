'''
Processes the Medsynth dataset to produce a file of the individual sentences by doctor/patient by conversation

'''

import csv
import re
import random

from helpers import process_medsynth

random.seed(42)

df = process_medsynth()

INPUT_CSV = "data/MedSynth_huggingface_final.csv"
BERT_OUT = "data/medsynth_utterances.csv"

sketchengine_lines = []
bert_rows = []

with open(INPUT_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

sampled_rows = random.sample(all_rows, k=int(len(all_rows) * 0.2))
print(f"Sampled {len(sampled_rows)} conversations from {len(all_rows)} total")


for conv_id, row in enumerate(sampled_rows, start=1):
    dialogue = row['Dialogue']

    # Split into turns using [patient] / [doctor] tags
    pattern = r'\[(patient|doctor)\]:\s*'
    parts = re.split(pattern, dialogue)
    # parts comes out as: ['', 'patient', 'text...', 'doctor', 'text...', ...]
    # so we zip them in pairs skipping the first empty string
    turns = [(parts[i], parts[i+1].strip()) for i in range(1, len(parts)-1, 2)]


print(df.head())
# Build CSV rows (full dataset)
for conv_id, row in df.iterrows():
    dialogue = row['Dialogue']
    pattern = r'\[(patient|doctor)\]:\s*'
    parts = re.split(pattern, dialogue)
    turns = [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]

    for turn_id, (speaker, text) in enumerate(turns, start=1):
        bert_rows.append({
            'conversation_id': conv_id,
            'turn_id': turn_id,
            'speaker': speaker,
            'text': text,
            'icd_chapter': row['ICD_chapter']
        })

# output utterances CSV
with open(BERT_OUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['conversation_id', 'turn_id', 'speaker', 'text', 'icd_chapter'])
    writer.writeheader()
    writer.writerows(bert_rows)

print(f"Done. {len(bert_rows)} turns processed.")
print(f"BERT CSV: {BERT_OUT}")