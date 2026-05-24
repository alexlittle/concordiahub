'''
Processes the Medsynth dataset to produce a file of the subjective and objective notes
Includes an UUID and the ICD-Chapter

'''

import uuid
from helpers import process_medsynth
from examples import config


df = process_medsynth()

df.insert(0, config.SAMPLE_UUID_COL, [str(uuid.uuid4()) for _ in range(len(df))])

df['scenario'] = df['Note'].str.split('**3.', regex=False).str[0]

# Extract uuid, dialog, icd_chapter
df_processed = df[[config.SAMPLE_UUID_COL, 'scenario', 'ICD_chapter']].copy()
df_processed.to_csv(config.MEDSYNTH_FULL_FILE, index=False)
