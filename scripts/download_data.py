import zipfile
import json
from pathlib import Path

from lexcite.config import KAGGLE_USERNAME, KAGGLE_KEY, RAW_DATA_DIR


def create_kaggle_json(username, key) -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_file = kaggle_dir / "kaggle.json"

    kaggle_dir.mkdir(exist_ok=True)
    if kaggle_file.exists():
        print(f"kaggle.json exists at {kaggle_file}")
        return

    with open(kaggle_file, "w") as f:
        json.dump(
            {"username": username, "key": key},
            f,
        )

    print(f"kaggle.json created at {kaggle_file}")


def download_kaggle_data(competition: str) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        raise ValueError("No KAGGLE_USERNAME or KAGGLE_KEY found in .env !")
    
    create_kaggle_json(KAGGLE_USERNAME, KAGGLE_KEY)
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    api.competition_download_files(
        competition=competition,
        path=str(RAW_DATA_DIR),
        quiet=False
    )
    print(f"Dataset downloaded to: {RAW_DATA_DIR}")

    zip_path = RAW_DATA_DIR / f"{competition}.zip"
    if zip_path.exists():
        print("Extracting dataset zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DATA_DIR)
        print(f"Dataset zip extracted to {RAW_DATA_DIR}")


if __name__ == "__main__":
    competition = "llm-agentic-legal-information-retrieval"
    download_kaggle_data(competition)
