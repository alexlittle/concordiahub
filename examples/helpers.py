"""
Helper functions for model development

"""

import pandas as pd
import re
import subprocess

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Mapping of Cleveland heart data column int values to their text names
HEART_COLUMN_MAPPINGS = {
    'sex': { 0: 'Female', 1: 'Male'},
    'chest pain type': { 1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal pain', 4: 'asymptomatic'},
    'fasting blood sugar': { 0: 'false', 1: 'true'},
    'resting ECG': { 0: 'normal', 1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy'},
    'exercise induced angina': { 0: 'no', 1: 'yes'},
    'slope of peak exercise': { 1: 'upsloping', 2: 'flat', 3: 'downsloping'},
    'thalassemia': { 3: 'normal', 6: 'fixed defect', 7: 'reversible defect'}
}

# mapping of ICD codes to their full names
ICD10_CHAPTERS = {
    "A": "Certain infectious and parasitic diseases",
    "B": "Certain infectious and parasitic diseases",
    "C": "Neoplasms",
    "D": "Neoplasms",
    "E": "Endocrine, nutritional and metabolic diseases",
    "F": "Mental, behavioral and neurodevelopmental disorders",
    "G": "Diseases of the nervous system",
    "H": "Diseases of eye/ear/adnexa/mastoid",
    "I": "Diseases of the circulatory system",
    "J": "Diseases of the respiratory system",
    "K": "Diseases of the digestive system",
    "L": "Diseases of the skin and subcutaneous tissue",
    "M": "Diseases of musculoskeletal system",
    "N": "Diseases of the genitourinary system",
    "O": "Pregnancy, childbirth and puerperium",
    "P": "Perinatal conditions",
    "Q": "Congenital malformations",
    "R": "Symptoms, signs, abnormal findings",
    "S": "Injury and poisoning",
    "T": "Injury and poisoning",
    "V": "External causes of morbidity",
    "W": "External causes of morbidity",
    "X": "External causes of morbidity",
    "Y": "External causes of morbidity",
    "Z": "Factors influencing health status",
    "U": "Special purposes"
}

# apply the ICD code mapping
def map_icd10_to_chapter(code):
    if pd.isna(code):
        return None
    return str(code).strip()[0].upper()

def check_gpu():
    try:
        subprocess.check_output('nvidia-smi')
        return "cuda"
    except Exception:
        return "cpu"

def safe_dirname(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def process_medsynth():
    # original medsynth link: https://huggingface.co/datasets/Ahmad0067/MedSynth/resolve/main/MedSynth_huggingface_final.csv
    df = pd.read_csv('./data/MedSynth_huggingface_final.csv')
    df = df.dropna(subset=["Dialogue"]).copy()
    df["ICD10_chapter"] = df["ICD10"].apply(map_icd10_to_chapter)
    df = df.rename(columns={" Note": "Note"})
    return df

def extract_spans_from_doc(doc) -> str:
    """Extracts noun chunks from a sentence."""
    spans = (
            [chunk.text.lower() for chunk in doc.noun_chunks] +
            [ent.text.lower() for ent in doc.ents]
    )
    seen = set()
    unique_spans = []
    for s in spans:
        # strip chars that will break feature name columns and brackets
        s = re.sub(r'[\[\]<>()\{\}]', '', s).strip()
        if ( s
                and len(s) >= 4 # must be at least 4 chars
                and not re.fullmatch(r'[\d\s,.\-]+', s) # can't be numerical only
                and s not in seen
        ):
            seen.add(s)
            unique_spans.append(s)
    return " | ".join(unique_spans)


def clean_text(text: str) -> str:
    # Strip formatting/punctuation/spaces before processing.
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*{1,2}', '', text)
    text = re.sub(r'^\s*[\*\-]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_confusion_matrix(y_true, y_pred, dataset, model):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"perf_results/cm_{dataset}_{model}.png", dpi=300)
    plt.show()

def save_class_report(report, dataset, model):
    with open(f"perf_results/classification_{dataset}_{model}.txt", "w") as f:
        f.write(report)