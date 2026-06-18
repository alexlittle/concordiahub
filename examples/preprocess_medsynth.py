'''
Processes the Medsynth dataset to produce a file of the subjective and objective notes
Includes an UUID and the ICD-Chapter

'''

import uuid
from helpers import process_medsynth
from examples import config


df = process_medsynth()

df.insert(0, config.INSTANCE_UUID_COL, [str(uuid.uuid4()) for _ in range(len(df))])

df['scenario'] = df['Note'].str.split('**3.', regex=False).str[0]

# Extract uuid, dialog, icd_chapter
df_processed = df[[config.INSTANCE_UUID_COL, 'scenario', 'ICD10_chapter']].copy()
df_processed = df_processed.rename(columns={"ICD10_chapter": "target"})

# update the duplicate columns
CHAPTER_MAP = {
    'B': 'A',
    'D': 'C',
    'T': 'S',
    'W': 'V',
    'X': 'V',
    'Y': 'V',
}
df_processed['target'] = df_processed['target'].replace(CHAPTER_MAP)

df_processed.to_csv(config.MEDSYNTH_FULL_FILE, index=False)
