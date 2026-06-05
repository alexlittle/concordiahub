import requests
import time
import pandas as pd
import re
import numpy as np
import subprocess
from pathlib import Path
from tqdm import tqdm
from typing import Union

ICD10_CODES = {
    "A": ("A00", "B99", "Certain infectious and parasitic diseases"),
    "B": ("A00", "B99", "Certain infectious and parasitic diseases"),
    "C": ("C00", "D49", "Neoplasms"),
    "D": ("C00", "D49", "Neoplasms"),
    "E": ("E00", "E89", "Endocrine, nutritional and metabolic diseases"),
    "F": ("F01", "F99", "Mental, behavioral and neurodevelopmental disorders"),
    "G": ("G00", "G99", "Diseases of the nervous system"),
    "H": ("H00", "H95", "Diseases of eye/ear/adnexa/mastoid"),
    "I": ("I00", "I99", "Diseases of the circulatory system"),
    "J": ("J00", "J99", "Diseases of the respiratory system"),
    "K": ("K00", "K95", "Diseases of the digestive system"),
    "L": ("L00", "L99", "Diseases of the skin and subcutaneous tissue"),
    "M": ("M00", "M99", "Diseases of musculoskeletal system"),
    "N": ("N00", "N99", "Diseases of the genitourinary system"),
    "O": ("O00", "O9A", "Pregnancy, childbirth and puerperium"),
    "P": ("P00", "P96", "Perinatal conditions"),
    "Q": ("Q00", "Q99", "Congenital malformations"),
    "R": ("R00", "R99", "Symptoms, signs, abnormal findings"),
    "S": ("S00", "T88", "Injury and poisoning"),
    "T": ("S00", "T88", "Injury and poisoning"),
    "V": ("V00", "Y99", "External causes of morbidity"),
    "W": ("V00", "Y99", "External causes of morbidity"),
    "X": ("V00", "Y99", "External causes of morbidity"),
    "Y": ("V00", "Y99", "External causes of morbidity"),
    "Z": ("Z00", "Z99", "Factors influencing health status"),
    "U": ("U00", "U85", "Special purposes")
}


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

def download_medsynth():
    url = "https://huggingface.co/datasets/Ahmad0067/MedSynth/resolve/main/MedSynth_huggingface_final.csv"
    download_file_if_needed(url, './data/MedSynth_huggingface_final.csv')
    return pd.read_csv('./data/MedSynth_huggingface_final.csv')


def process_medsynth():
    df = download_medsynth()
    df = df.dropna(subset=["Dialogue"]).copy()
    df["ICD10_chapter"] = df["ICD10"].apply(map_icd10_to_chapter)
    df = df.rename(columns={" Note": "Note"})

    return df


def download_file_if_needed(
    url: str,
    dest_path: Union[str, Path],
    chunk_size: int = 1024 * 1024,
    max_retries: int = 5,
    timeout: int = 30,
    show_progress: bool = True,
    resume: bool = True,
):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"File already exists: {dest_path}")
        return dest_path

    headers = {}
    try:
        head = requests.head(url, allow_redirects=True, timeout=timeout)
        head.raise_for_status()
        total_size = int(head.headers.get("Content-Length", 0))
        accept_ranges = head.headers.get("Accept-Ranges", "").lower() == "bytes"
    except Exception:
        total_size = 0
        accept_ranges = True

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    downloaded = temp_path.stat().st_size if temp_path.exists() else 0

    if resume and downloaded > 0 and accept_ranges:
        headers["Range"] = f"bytes={downloaded}-"

    pbar = tqdm(
        total=None if total_size == 0 else (total_size - downloaded),
        unit="B",
        unit_scale=True,
        desc=dest_path.name,
        initial=downloaded
    ) if show_progress else None

    attempt = 0
    while attempt < max_retries:
        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
                if r.status_code in (200, 206):
                    mode = "ab" if "Range" in headers else "wb"

                    if "Range" in headers and r.status_code == 200:
                        downloaded = 0
                        mode = "wb"
                        if pbar:
                            pbar.reset(total=total_size or None)
                            pbar.n = 0
                            pbar.refresh()

                    with open(temp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                if pbar:
                                    pbar.update(len(chunk))

                    temp_path.replace(dest_path)
                    if pbar:
                        pbar.close()
                    print(f"Downloaded to: {dest_path}")
                    return dest_path

                else:
                    r.raise_for_status()

        except Exception as e:
            attempt += 1
            if pbar:
                pbar.set_postfix_str(f"retry {attempt}/{max_retries}")
            time.sleep(min(2 ** attempt, 10))

            downloaded = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {}
            if resume and downloaded > 0 and accept_ranges:
                headers["Range"] = f"bytes={downloaded}-"

            if attempt >= max_retries:
                if pbar:
                    pbar.close()
                raise RuntimeError(f"Failed after {max_retries} attempts: {e}") from e