"""
Download / load the UCI Student Performance dataset (math course: student-mat.csv).

WHY this file exists: the notebook should run on a fresh clone with one command,
and a reader should be able to see exactly where the data came from.

Source: P. Cortez and A. Silva (2008), UCI Machine Learning Repository.
Run:  python data/download_data.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "student-mat.csv"

# Tried in order. The first two are the official UCI locations (zip files).
# The last is a public GitHub copy of the same CSV, used only if UCI is unreachable.
ZIP_SOURCES = [
    "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
]
CSV_FALLBACK = (
    "https://raw.githubusercontent.com/avehtari/ROS-Examples/"
    "master/Student/data/student-mat.csv"
)

EXPECTED_SHAPE = (395, 33)  # published size of the math dataset


def _find_csv_in_zip(zf):
    """Return the bytes of student-mat.csv from a zip, looking inside nested zips too.
    WHY nested: some UCI downloads wrap the data in a second zip file."""
    for name in zf.namelist():
        if name.endswith("student-mat.csv"):
            return zf.read(name)
    for name in zf.namelist():
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(zf.read(name))) as inner:
                found = _find_csv_in_zip(inner)
                if found is not None:
                    return found
    return None


def _fetch(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def download(zip_sources=ZIP_SOURCES, csv_fallback=CSV_FALLBACK, verbose=True):
    """Fetch the CSV, save it to data/student-mat.csv, and return the source used."""
    for url in zip_sources:
        try:
            with zipfile.ZipFile(io.BytesIO(_fetch(url))) as zf:
                content = _find_csv_in_zip(zf)
            if content is not None:
                CSV_PATH.write_bytes(content)
                return url
        except Exception as err:  # network down, bad zip, 404 ... try the next source
            if verbose:
                print(f"  could not use {url}: {type(err).__name__}")
    CSV_PATH.write_bytes(_fetch(csv_fallback))
    return csv_fallback


def load_student_mat(verbose=True):
    """Load the dataset as a DataFrame, downloading it first if it is not on disk.
    Note sep=';' : this file is semicolon-separated, not comma-separated."""
    if not CSV_PATH.exists():
        source = download(verbose=verbose)
        if verbose:
            print(f"Downloaded student-mat.csv from: {source}")
    df = pd.read_csv(CSV_PATH, sep=";")
    # WHY validate: fail loudly if we got an HTML error page or the wrong file.
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape {df.shape}, expected {EXPECTED_SHAPE}")
    return df


if __name__ == "__main__":
    data = load_student_mat()
    print("OK:", data.shape, "->", CSV_PATH)
