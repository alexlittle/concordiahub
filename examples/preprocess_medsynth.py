'''
Processes the Medsynth dataset to produce a file of the individual sentences by doctor/patient by conversation

'''

import re
import random
import uuid
import pandas as pd
from helpers import process_medsynth
from examples import config
random.seed(42)

df = process_medsynth()

df.insert(0, config.SAMPLE_UUID_COL, [str(uuid.uuid4()) for _ in range(len(df))])
df.index = df[config.SAMPLE_UUID_COL]

# File 1: conversations - uuid, dialog, icd_chapter
df = df[[config.SAMPLE_UUID_COL, 'Dialogue', 'ICD_chapter']]
df.to_csv(config.MEDSYNTH_FULL_FILE, index=False)


# File 2: utterances - uuid, turn_id, speaker, text, icd_chapter
utterance_rows = []
for conv_uuid, row in df.iterrows():
    pattern = r'\[(patient|doctor)\]:\s*'
    parts = re.split(pattern, row['Dialogue'])
    turns = [(parts[i], parts[i+1].strip()) for i in range(1, len(parts)-1, 2)]
    for turn_id, (speaker, text) in enumerate(turns, start=1):
        utterance_rows.append({
            config.SAMPLE_UUID_COL: conv_uuid,
            'turn_id': turn_id,
            'speaker': speaker,
            'text': text,
            'icd_chapter': row['ICD_chapter']
        })

utterances_df = pd.DataFrame(utterance_rows)
utterances_df.to_csv(config.MEDSYNTH_UTTERANCES_FILE, index=False)

print(f"Done. {len(utterances_df)} turns processed.")