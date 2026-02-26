import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"

load_dotenv(ROOT_DIR / ".env")


def download_model_hf(repo_id: str, filename: str):
    print("Downloading GGUF model from Hugging Face...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(MODEL_DIR),
        token=os.getenv("HF_TOKEN")
    )

    print(f"Model downloaded: {model_path}")


if __name__ == "__main__":
    repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    download_model_hf(repo_id, filename)


