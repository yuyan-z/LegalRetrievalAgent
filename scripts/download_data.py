import zipfile
import json
from pathlib import Path

from uretriever.configs.api_config import get_env


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


def download_kaggle_data(competition: str, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)

    kaggle_username = get_env("KAGGLE_USERNAME")
    kaggle_key = get_env("KAGGLE_KEY")

    create_kaggle_json(kaggle_username, kaggle_key)
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    api.competition_download_files(
        competition=competition,
        path=str(save_path),
        quiet=False
    )
    print(f"Dataset downloaded to: {save_path}")

    zip_path = save_path / f"{competition}.zip"
    if zip_path.exists():
        print("Extracting dataset zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(save_path)
        print(f"Dataset zip extracted to {save_path}")


if __name__ == "__main__":
    from uretriever.configs.path_config import DATA_DIR

    competition = "llm-agentic-legal-information-retrieval"
    save_path = DATA_DIR / "raw"

    download_kaggle_data(competition, save_path)
